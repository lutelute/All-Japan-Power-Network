"""Hardware and solver detection for adaptive UC solver strategy selection.

Detects CPU core count, available RAM, OS/architecture, and available
MIP solvers at runtime. Gracefully handles missing ``psutil`` dependency
by falling back to conservative defaults (1 physical core, 2 GB RAM).

Usage::

    from src.uc.hardware_detector import detect_hardware, detect_available_solvers

    profile = detect_hardware()
    print(f"Cores: {profile.physical_cores}, RAM: {profile.available_ram_gb:.1f} GB")
    print(f"Solvers: {profile.available_solvers}")
"""

import os
import platform
from dataclasses import dataclass, field
from typing import List

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Guard psutil import — it is an optional dependency
try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# Conservative fallback defaults when psutil is unavailable
_DEFAULT_PHYSICAL_CORES = 1
_DEFAULT_LOGICAL_CORES = 1
_DEFAULT_AVAILABLE_RAM_GB = 2.0
_DEFAULT_TOTAL_RAM_GB = 2.0


@dataclass
class HardwareProfile:
    """Detected hardware capabilities for solver strategy selection.

    Attributes:
        physical_cores: Number of physical CPU cores. Falls back to 1
            when detection fails (e.g., virtualised environments where
            ``cpu_count(logical=False)`` returns ``None``).
        logical_cores: Number of logical CPU cores (includes
            hyper-threading). Falls back to ``physical_cores`` when
            detection fails.
        available_ram_gb: Currently available (free) RAM in gigabytes.
            Falls back to 2.0 GB when ``psutil`` is not installed.
        total_ram_gb: Total installed RAM in gigabytes. Falls back to
            2.0 GB when ``psutil`` is not installed.
        available_solvers: List of PuLP solver names detected as
            available on the system (e.g., ``['HiGHS_CMD', 'PULP_CBC_CMD']``).
        os_name: Operating system name (e.g., ``'Darwin'``, ``'Linux'``).
        architecture: CPU architecture string (e.g., ``'arm64'``,
            ``'x86_64'``).
    """

    physical_cores: int = _DEFAULT_PHYSICAL_CORES
    logical_cores: int = _DEFAULT_LOGICAL_CORES
    available_ram_gb: float = _DEFAULT_AVAILABLE_RAM_GB
    total_ram_gb: float = _DEFAULT_TOTAL_RAM_GB
    available_solvers: List[str] = field(default_factory=list)
    os_name: str = ""
    architecture: str = ""

    def __post_init__(self) -> None:
        """Validate hardware profile parameters."""
        if self.physical_cores < 1:
            raise ValueError(
                f"physical_cores must be >= 1, got {self.physical_cores}"
            )
        if self.logical_cores < 1:
            raise ValueError(
                f"logical_cores must be >= 1, got {self.logical_cores}"
            )
        if self.available_ram_gb < 0:
            raise ValueError(
                f"available_ram_gb must be non-negative, got {self.available_ram_gb}"
            )
        if self.total_ram_gb < 0:
            raise ValueError(
                f"total_ram_gb must be non-negative, got {self.total_ram_gb}"
            )


def detect_available_solvers() -> List[str]:
    """Detect which PuLP MIP solvers are available on the system.

    Uses ``pulp.listSolvers(onlyAvailable=True)`` to enumerate solvers
    that are both installed and accessible. Falls back to an empty list
    if PuLP is not installed or solver detection fails.

    Returns:
        Sorted list of available solver name strings
        (e.g., ``['HiGHS_CMD', 'PULP_CBC_CMD']``).
    """
    try:
        import pulp

        solvers = pulp.listSolvers(onlyAvailable=True)
        logger.debug("Detected available solvers: %s", solvers)
        return sorted(solvers)
    except Exception:
        logger.warning("Failed to detect available solvers via PuLP")
        return []


def _detect_cpu_cores() -> tuple:
    """Detect physical and logical CPU core counts.

    Returns:
        Tuple of ``(physical_cores, logical_cores)``. Uses ``psutil``
        when available, otherwise falls back to ``os.cpu_count()`` for
        logical cores and conservative default for physical cores.
    """
    if _PSUTIL_AVAILABLE:
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)

        # cpu_count(logical=False) can return None on some virtualised
        # environments (e.g., containers, certain cloud VMs)
        if physical is None:
            logger.warning(
                "psutil.cpu_count(logical=False) returned None; "
                "defaulting to %d physical core(s)",
                _DEFAULT_PHYSICAL_CORES,
            )
            physical = _DEFAULT_PHYSICAL_CORES

        if logical is None:
            logger.warning(
                "psutil.cpu_count(logical=True) returned None; "
                "defaulting to physical core count (%d)",
                physical,
            )
            logical = physical

        return physical, logical

    # Fallback without psutil: os.cpu_count() returns logical cores only
    os_cores = os.cpu_count()
    if os_cores is not None and os_cores >= 1:
        return _DEFAULT_PHYSICAL_CORES, os_cores

    return _DEFAULT_PHYSICAL_CORES, _DEFAULT_LOGICAL_CORES


def _detect_ram() -> tuple:
    """Detect available and total RAM in gigabytes.

    Returns:
        Tuple of ``(available_ram_gb, total_ram_gb)``. Uses ``psutil``
        when available, otherwise returns conservative defaults.
    """
    if _PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)
            return available_gb, total_gb
        except Exception:
            logger.warning("Failed to read memory info from psutil")
            return _DEFAULT_AVAILABLE_RAM_GB, _DEFAULT_TOTAL_RAM_GB

    return _DEFAULT_AVAILABLE_RAM_GB, _DEFAULT_TOTAL_RAM_GB


def detect_hardware() -> HardwareProfile:
    """Detect hardware capabilities for solver strategy selection.

    Probes CPU core count, available/total RAM, OS/architecture, and
    available MIP solvers. Safe to call on any platform — gracefully
    degrades when ``psutil`` is not installed or detection APIs return
    unexpected values.

    Returns:
        A ``HardwareProfile`` instance populated with detected (or
        conservative fallback) values.

    Example::

        profile = detect_hardware()
        print(f"Physical cores: {profile.physical_cores}")
        print(f"Available RAM: {profile.available_ram_gb:.1f} GB")
        print(f"Solvers: {profile.available_solvers}")
    """
    logger.info("Detecting hardware capabilities...")

    if not _PSUTIL_AVAILABLE:
        logger.warning(
            "psutil not available; using conservative defaults "
            "(cores=%d, ram=%.1f GB)",
            _DEFAULT_PHYSICAL_CORES,
            _DEFAULT_AVAILABLE_RAM_GB,
        )

    physical_cores, logical_cores = _detect_cpu_cores()
    available_ram_gb, total_ram_gb = _detect_ram()
    available_solvers = detect_available_solvers()
    os_name = platform.system()
    architecture = platform.machine()

    profile = HardwareProfile(
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        available_ram_gb=available_ram_gb,
        total_ram_gb=total_ram_gb,
        available_solvers=available_solvers,
        os_name=os_name,
        architecture=architecture,
    )

    logger.info(
        "Hardware detected: %d physical cores, %d logical cores, "
        "%.1f GB available RAM (%.1f GB total), OS=%s, arch=%s, "
        "solvers=%s",
        profile.physical_cores,
        profile.logical_cores,
        profile.available_ram_gb,
        profile.total_ram_gb,
        profile.os_name,
        profile.architecture,
        profile.available_solvers,
    )

    return profile
