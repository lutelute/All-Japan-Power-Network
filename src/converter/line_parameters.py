"""Japanese transmission line electrical parameter reference table.

Loads standard reference values for Japanese transmission lines from
``config/line_types.yaml`` and provides runtime conversion of susceptance
B (S/km) to capacitance c (nF/km) for pandapower compatibility.

The conversion is frequency-dependent:
    c_nf_per_km = b_s_per_km / (2 * pi * f_hz) * 1e9

East Japan (Hokkaido, Tohoku, Tokyo) uses 50 Hz;
West Japan (Chubu and westward) uses 60 Hz.

Usage::

    from src.converter.line_parameters import get_line_parameters

    params = get_line_parameters(275, 50)
    # {'r_ohm_per_km': 0.028, 'x_ohm_per_km': 0.325,
    #  'c_nf_per_km': 12.25, 'max_i_ka': 2.0, ...}
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Supported system frequencies in Japan
VALID_FREQUENCIES_HZ = (50, 60)

# Default path to the line types configuration file
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "line_types.yaml"

# Module-level cache for loaded YAML data
_line_types_cache: Optional[Dict[int, Dict[str, Any]]] = None


def _load_line_types(config_path: Optional[Path] = None) -> Dict[int, Dict[str, Any]]:
    """Load transmission line type parameters from YAML configuration.

    Args:
        config_path: Path to ``line_types.yaml``. Defaults to
            ``config/line_types.yaml`` relative to the project root.

    Returns:
        Dictionary keyed by voltage class (int kV) with parameter dicts.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    global _line_types_cache

    path = config_path or _DEFAULT_CONFIG_PATH

    if _line_types_cache is not None and config_path is None:
        return _line_types_cache

    if not path.exists():
        raise FileNotFoundError(
            f"Line types configuration not found: {path}"
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict at top level of {path}, got {type(raw).__name__}")

    line_types: Dict[int, Dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, int):
            logger.warning("Skipping non-integer key in line_types.yaml: %s", key)
            continue
        if not isinstance(value, dict):
            logger.warning("Skipping non-dict value for voltage %d kV", key)
            continue
        line_types[key] = value

    logger.info(
        "Loaded line parameters for %d voltage classes: %s kV",
        len(line_types),
        sorted(line_types.keys(), reverse=True),
    )

    if config_path is None:
        _line_types_cache = line_types

    return line_types


def b_to_c_nf_per_km(b_s_per_km: float, f_hz: float) -> float:
    """Convert susceptance B (S/km) to capacitance c (nF/km).

    pandapower's ``create_line_from_parameters()`` requires capacitance in
    nF/km, while engineering reference data typically uses susceptance in
    S/km. This conversion is frequency-dependent.

    Formula::

        c_nf_per_km = b_s_per_km / (2 * pi * f_hz) * 1e9

    Args:
        b_s_per_km: Line susceptance in Siemens per kilometer.
        f_hz: System frequency in Hertz (50 or 60).

    Returns:
        Line capacitance in nanofarads per kilometer.

    Raises:
        ValueError: If frequency is not positive.
    """
    if f_hz <= 0:
        raise ValueError(f"Frequency must be positive, got {f_hz} Hz")

    return b_s_per_km / (2.0 * math.pi * f_hz) * 1e9


def get_line_parameters(
    voltage_kv: float,
    f_hz: float,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Get electrical parameters for a Japanese transmission line type.

    Looks up standard reference values by voltage class and converts
    susceptance B (S/km) to capacitance c (nF/km) based on the given
    system frequency.

    Args:
        voltage_kv: Nominal voltage in kilovolts (e.g., 500, 275, 154, 66).
        f_hz: System frequency in Hertz (50 for east Japan, 60 for west).
        config_path: Optional override for the config file path.

    Returns:
        Dictionary containing:
            - ``r_ohm_per_km``: Resistance (Ohm/km)
            - ``x_ohm_per_km``: Reactance (Ohm/km)
            - ``b_s_per_km``: Original susceptance (S/km) from reference data
            - ``c_nf_per_km``: Capacitance (nF/km) converted for pandapower
            - ``max_i_ka``: Maximum current rating (kA)
            - ``conductor``: Conductor specification string
            - ``circuits``: Number of circuits
            - ``grid_class``: Grid hierarchy classification

    Raises:
        ValueError: If the voltage class is not found in the reference table
            or the frequency is invalid.
        FileNotFoundError: If the configuration file is missing.
    """
    if f_hz not in VALID_FREQUENCIES_HZ:
        raise ValueError(
            f"Frequency must be one of {VALID_FREQUENCIES_HZ}, got {f_hz} Hz"
        )

    line_types = _load_line_types(config_path)

    # Look up by rounded integer voltage
    voltage_key = int(round(voltage_kv))

    if voltage_key not in line_types:
        available = sorted(line_types.keys(), reverse=True)
        raise ValueError(
            f"No line parameters for {voltage_key} kV. "
            f"Available voltage classes: {available}"
        )

    raw_params = line_types[voltage_key]

    # Extract required electrical parameters
    r_ohm = raw_params["r_ohm_per_km"]
    x_ohm = raw_params["x_ohm_per_km"]
    b_s = raw_params["b_s_per_km"]
    max_i = raw_params["max_i_ka"]

    # Convert susceptance to capacitance for pandapower
    c_nf = b_to_c_nf_per_km(b_s, f_hz)

    result: Dict[str, Any] = {
        "r_ohm_per_km": r_ohm,
        "x_ohm_per_km": x_ohm,
        "b_s_per_km": b_s,
        "c_nf_per_km": c_nf,
        "max_i_ka": max_i,
        "conductor": raw_params.get("conductor", ""),
        "circuits": raw_params.get("circuits", 1),
        "grid_class": raw_params.get("grid_class", ""),
    }

    logger.debug(
        "%d kV @ %d Hz: R=%.4f, X=%.4f, c_nf=%.2f, max_I=%.1f kA",
        voltage_key,
        int(f_hz),
        r_ohm,
        x_ohm,
        c_nf,
        max_i,
    )

    return result


def get_line_parameters_safe(
    voltage_kv: float,
    f_hz: float,
    config_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Get line parameters with fallback to nearest voltage class.

    Unlike :func:`get_line_parameters`, this function does not raise on
    unrecognized voltage classes. Instead, it falls back to the nearest
    available voltage class and logs a warning.

    Args:
        voltage_kv: Nominal voltage in kilovolts.
        f_hz: System frequency in Hertz (50 or 60).
        config_path: Optional override for the config file path.

    Returns:
        Parameter dictionary (see :func:`get_line_parameters`), or None
        if no suitable fallback exists.
    """
    try:
        return get_line_parameters(voltage_kv, f_hz, config_path)
    except ValueError:
        pass

    # Attempt nearest voltage class fallback
    try:
        line_types = _load_line_types(config_path)
    except (FileNotFoundError, ValueError):
        logger.error("Cannot load line type configuration for fallback")
        return None

    if not line_types:
        return None

    voltage_key = int(round(voltage_kv))
    nearest_kv = min(line_types.keys(), key=lambda k: abs(k - voltage_key))

    logger.warning(
        "No exact parameters for %d kV; falling back to nearest: %d kV",
        voltage_key,
        nearest_kv,
    )

    return get_line_parameters(float(nearest_kv), f_hz, config_path)


def get_available_voltage_classes(
    config_path: Optional[Path] = None,
) -> list:
    """Return a sorted list of available voltage classes (kV).

    Args:
        config_path: Optional override for the config file path.

    Returns:
        List of voltage classes in descending order (e.g., [500, 275, ...]).
    """
    line_types = _load_line_types(config_path)
    return sorted(line_types.keys(), reverse=True)


def clear_cache() -> None:
    """Clear the module-level line types cache.

    Useful for testing or when the configuration file has been modified
    and needs to be reloaded.
    """
    global _line_types_cache
    _line_types_cache = None
